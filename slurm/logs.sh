JOB_ID=$(sacct -XPno jobid | tail -1); cat logs/DAS-ILU-"$JOB_ID"*
