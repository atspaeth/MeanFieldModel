from mfsupport import firing_rates, become_worker

if __name__ == "__main__":
    become_worker("sim", lambda job: firing_rates(**job.params))
