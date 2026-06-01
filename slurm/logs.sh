JOB_ID=$(sacct -XPno jobid | tail -1); tail --follow logs/DAS-ILU-"$JOB_ID"*
