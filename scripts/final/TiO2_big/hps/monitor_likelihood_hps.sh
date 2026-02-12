#!/bin/bash
# Monitor BNN_Forces_Likelihood HPS jobs (LRT and RAD)

echo "=== Job Status ==="
qstat -u g15farris 2>/dev/null | grep -E "hps_lrt|hps_rad" || echo "No likelihood HPS jobs running"

echo ""
echo "=== LRT HPS (last 30 lines) ==="
tail -30 /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_lrt_likelihood.out 2>/dev/null || echo "No output yet"

echo ""
echo "=== RAD HPS (last 30 lines) ==="
tail -30 /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_rad_likelihood.out 2>/dev/null || echo "No output yet"

echo ""
echo "=== Optuna DB status (TiO2_big) ==="
if [ -f /home/g15farris/bin/bayesaenet/bnn_aenet/results/TiO2_big/bnn_lrt_forces_likelihood.db ]; then
    echo "LRT trials: $(sqlite3 /home/g15farris/bin/bayesaenet/bnn_aenet/results/TiO2_big/bnn_lrt_forces_likelihood.db "SELECT COUNT(*) FROM trials;" 2>/dev/null || echo "?")"
fi
if [ -f /home/g15farris/bin/bayesaenet/bnn_aenet/results/TiO2_big/bnn_rad_forces_likelihood.db ]; then
    echo "RAD trials: $(sqlite3 /home/g15farris/bin/bayesaenet/bnn_aenet/results/TiO2_big/bnn_rad_forces_likelihood.db "SELECT COUNT(*) FROM trials;" 2>/dev/null || echo "?")"
fi

echo ""
echo "=== Recent errors (if any) ==="
tail -5 /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_lrt_likelihood.err 2>/dev/null
tail -5 /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_rad_likelihood.err 2>/dev/null
