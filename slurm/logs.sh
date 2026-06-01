JOB_ID=$(sacct -XPno jobid | tail -1); tail --follow logs/DAS_ILU-"$JOB_ID"*
