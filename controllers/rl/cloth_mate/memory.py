import h5py
from filelock import FileLock
import numpy as np
import time

class Memory:

    def __init__(self, memory_fields=[]):
        self.data = {}
        for memory_field in memory_fields:
            self.data[memory_field] = []
        self.length = 0

    @staticmethod
    def concat(memories):
        output = Memory()
        for memory in memories:
            for key in memory.data:
                if key not in output.data:
                    output.data[key] = []
                output.data[key].extend(memory.data[key])
        return output

    def clear(self):
        for key in list(self.data.keys()):
            if isinstance(self.data[key], list):
                del self.data[key][:]
            else:
                del self.data[key]

    def print_length(self):
        output = "[Memory] "
        for key in self.data:
            output += f" {key}: {len(self.data[key])} |"
        print(output)
        
    def assert_length(self):
        excluded_keys = {'data_dict', 'cloth_category', 'cloth_instance'}
        key_lens = []
        mismatched_keys = []

        for key, item in self.data.items():
            if key in excluded_keys:
                continue
            if isinstance(item, (list, np.ndarray)):
                key_lens.append((key, len(item)))
            else:
                raise TypeError(f"Item for key '{key}' is not a list or array.")

        if not key_lens: 
            return

        lengths = [length for _, length in key_lens]
        unique_lengths = np.unique(lengths)

        if len(unique_lengths) == 1:
            self.length = unique_lengths[0]
            return 

        for key, length in key_lens:
            print(f"Key: {key}, Length: {length}")

        if mismatched_keys:
            print("Mismatched keys and lengths:")
            for key, length in mismatched_keys:
                print(f"Key: {key}, Length: {length}")

        min_length = min(lengths)
        for key, _ in key_lens:
            if len(self.data[key]) != min_length:
                self.data[key] = self.data[key][:min_length]

        self.length = min_length

    def __len__(self):
        return self.length

    def add_value(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def keys(self):
        return [key for key in self.data]

    def count(self):
        return len(self.data['observations'])

    def done(self):
        if len(self.data['is_terminal']) == 0:
            return False
        return self.data['is_terminal'][-1]

    def get_data(self):
        return self.data

    def check_error(self):
        try:
            count = len(self)
            assert len(self.data['max_coverage']) == count
            assert len(self.data['preaction_coverage']) == count
            assert len(self.data['postaction_coverage']) == count
            return True
        except:
            return False

    def exist_instance(self, hdf5_path):
        cloth_instance = self.data['cloth_instance'][0].split('_')[0]
        with FileLock(hdf5_path + ".lock"):
            with h5py.File(hdf5_path, 'a') as file:
                if cloth_instance in file:
                    return True
                else:
                    return False

    def dump(self, hdf5_path):
        print("[Memory] Dumping memory to", hdf5_path)
        while True:
            try:
                with FileLock(hdf5_path + ".lock"):
                    with h5py.File(hdf5_path, 'a') as file:
                        if 'count' not in file.attrs.keys():
                            file.attrs['count'] = 0
                        cloth_instance = self.data['cloth_instance'].split('_')[0]
                        cloth_category = self.data['cloth_category']

                        for step in range(len(self)):
                            group = file.create_group(f'{cloth_category}/{cloth_instance}/{file.attrs["count"]:09d}')
                            file.attrs['count'] += 1
                            for key, value in self.data.items():
                                if key in ['cloth_instance', 'cloth_category']:
                                    continue
                                try:
                                    step_value = value[step]
                                    if type(step_value) == float or type(step_value) == np.float64 or\
                                            type(step_value) == np.int32 or type(step_value) == int or\
                                                type(step_value) == np.int64 or type(step_value) == str:
                                        group.attrs[key] = step_value 
                                    elif type(step_value) == list:
                                        subgroup = group.create_group(key)  
                                        for i, item in enumerate(step_value):
                                            subgroup.create_dataset(name=f'{i:09d}', data=item, compression='gzip')
                                    else:
                                        group.create_dataset(name=key, data=step_value, compression='gzip')
                                except Exception as e:
                                    print(f"[Memory] Dump [{key}] Error:", e)
                                    print(value)
                                    exit()
                                    time.sleep(0.1)
                return 
            except Exception as e:
                print("[Memory] Dump error:", e)
                time.sleep(0.1) 
                pass