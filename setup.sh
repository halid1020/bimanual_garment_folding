# Usage:  source ./setup.sh          # env only
#         source ./setup.sh xarm     # env + configure the wired NIC for the Lite 6 arms
_MP_SETUP_FLAG="${1:-}"   # captured first: sourcing softgym's setup.sh can clobber $1

conda activate magpie
if [ -d "../softgym" ]; then
  cd ../softgym
  . ./setup.sh
else
  echo "Directory ../softgym does not exist. Skipping."
fi

cd ../bimanual_garment_folding

export PYTHONPATH=${PWD}:$PYTHONPATH
export MP_FOLD_PATH=${PWD}
export REAL_ROBOT_PATH="${PWD}/real_robot"

# ---------------------------------------------------------------------------
# Dual xArm Lite 6 cell
# ---------------------------------------------------------------------------
# The control boxes sit on a private wired subnet and do NOT serve DHCP, so a
# DHCP-configured ethernet port never gets an address and every connection fails
# with "connect socket failed". `source ./setup.sh xarm` gives the wired NIC a
# static address on that subnet. ipv4.never-default keeps WiFi as the default
# route, so internet access is unaffected.
export XARM_LEFT_IP="${XARM_LEFT_IP:-192.168.1.155}"
export XARM_RIGHT_IP="${XARM_RIGHT_IP:-192.168.1.170}"

xarm_net_check() {
  local ip rc=0
  for ip in "$XARM_LEFT_IP" "$XARM_RIGHT_IP"; do
    if ping -c1 -W1 "$ip" >/dev/null 2>&1; then
      echo "[xarm]   $ip  reachable"
    else
      echo "[xarm]   $ip  NO RESPONSE  (power / cable / switch / wrong IP?)"
      rc=1
    fi
  done
  return $rc
}

xarm_net_setup() {
  local con="${XARM_CON_NAME:-xarm-lan}"
  local host_cidr="${XARM_HOST_IP:-192.168.1.100/24}"
  local nic="${XARM_NIC:-}"

  # Already on a directly-connected link? Then don't touch anything (and don't
  # trigger a sudo prompt). "via <gw>" means the packets would be routed off-box.
  if ! ip route get "$XARM_LEFT_IP" 2>/dev/null | head -1 | grep -q ' via '; then
    echo "[xarm] $XARM_LEFT_IP already directly routable - no network change needed."
    xarm_net_check
    return 0
  fi

  if [ -z "$nic" ]; then
    nic=$(nmcli -t -f DEVICE,TYPE device 2>/dev/null \
          | awk -F: '$2=="ethernet"{print $1; exit}')
  fi
  if [ -z "$nic" ]; then
    echo "[xarm] No ethernet device found. Set XARM_NIC=<iface> and re-run."
    return 1
  fi

  echo "[xarm] Configuring $nic -> $host_cidr (profile '$con'); sudo may prompt."
  if nmcli -t -f NAME connection show 2>/dev/null | grep -qx -- "$con"; then
    sudo nmcli connection modify "$con" \
         connection.interface-name "$nic" ipv4.method manual \
         ipv4.addresses "$host_cidr" ipv4.never-default yes ipv6.method ignore || return 1
  else
    sudo nmcli connection add type ethernet ifname "$nic" con-name "$con" \
         ipv4.method manual ipv4.addresses "$host_cidr" \
         ipv4.never-default yes ipv6.method ignore || return 1
  fi
  sudo nmcli connection up "$con" || return 1

  xarm_net_check
  echo "[xarm] Next: python real_robot/test/test_xarm_lite6_bringup.py --info-only --arm both"
}

case "$_MP_SETUP_FLAG" in
  xarm|--xarm|lite6|--lite6) xarm_net_setup ;;
esac
unset _MP_SETUP_FLAG
