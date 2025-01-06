#!/bin/bash

if [ "$1" = "sillok" ]; then
    python3 sillok_crawler.py
elif [ "$1" = "sji" ]; then
    echo "Not implemented yet."
else
    echo "Invalid argument or no argument provided."
fi