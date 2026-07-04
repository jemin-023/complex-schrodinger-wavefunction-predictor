#!/bin/bash
PID=$(pgrep -f "python train.py" | head -n 1)
if [ -z "$PID" ]; then
    echo "Error: No running train.py process found!"
    exit 1
fi

echo "Found train.py process with PID: $PID"
echo "Attaching GDB to save checkpoint.pkl..."

sudo gdb -p $PID \
  -ex 'call (int) PyRun_SimpleString("import gc, pickle; [pickle.dump(obj.params, open(\"checkpoint.pkl\", \"wb\")) for obj in gc.get_objects() if type(obj).__name__ == \"TrainState\"]")' \
  -ex 'detach' \
  -ex 'quit'

if [ -f "checkpoint.pkl" ]; then
    echo "SUCCESS: checkpoint.pkl has been saved."
else
    echo "FAILED: checkpoint.pkl was not created."
fi
