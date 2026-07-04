#!/bin/bash
PID=$(pgrep -f "python train.py" | head -n 1)
if [ -z "$PID" ]; then
    echo "Error: No running train.py process found!"
    exit 1
fi

echo "Found train.py process with PID: $PID"
echo "Attaching GDB to save checkpoint.pkl..."

sudo gdb -p $PID \
  -ex 'call (int) PyRun_SimpleString("import os, gc, pickle; os.makedirs(\"logs\", exist_ok=True); [pickle.dump(obj.params, open(\"logs/checkpoint.pkl\", \"wb\")) for obj in gc.get_objects() if type(obj).__name__ == \"TrainState\"]")' \
  -ex 'detach' \
  -ex 'quit'

if [ -f "logs/checkpoint.pkl" ]; then
    echo "SUCCESS: logs/checkpoint.pkl has been saved."
else
    echo "FAILED: logs/checkpoint.pkl was not created."
fi
