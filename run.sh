#!/bin/sh

# We can force output to be flushed to the log using the -u flag
python -u transfer_ko.py --embedding-dim 128 --hidden-dim 32 --epoches 10 -s True --is-all True --save-path './checkpoint/128-32-10.pt' >> log 2>&1
