#!/bin/bash
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8888 a.pinggy.io > pinggy.log 2>&1 &
