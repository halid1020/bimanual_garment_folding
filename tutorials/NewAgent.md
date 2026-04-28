## 4. Adding a New Agent / Controller

1. **Create the Controller:**
   Add your new controller script inside the `controllers` folder. Ensure it adheres to the `Agent` or `TrainableAgent` interface defined in the `actoris_harena` package. You can refer to the existing controllers in that folder for examples.
2. **Register the Controller:**
   Open `registration/agent.py` and add your new agent to the `register_agents` function. This makes it available to the training script.