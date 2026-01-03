#!/bin/bash

cd /root/CHIKI_AI_GOGH
uvicorn main:app --host 127.0.0.1 --port 8000 > server.log 2>&1