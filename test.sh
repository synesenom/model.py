#!/bin/bash
# Performs specific test on all distributions.

TEST="mle-fit"
while getopts "t:h" opt
do
	case $opt in
	t)
		TEST="$OPTARG";
		;;
	\?)
		echo "Invalid option: -$OPTARG";
		;;
	h)
		printf "Runs given test on all distributions."
		printf "usage:\\ntest.sh [options]\n"
		printf "options:\n"
		printf "  -h     print help menu.\n"
		printf "  -t     test type ([mle-fit], ks-fit, aic-ms, bic-ms, ks-ms).\n"
		exit
		;;
	esac
done

# run all tests
for d in exponential \
		 lognormal \
		 normal \
		 poisson \
		 shifted-power-law \
		 truncated-power-law \
		 weibull; do
	./model.py --test-${TEST} $d
done
