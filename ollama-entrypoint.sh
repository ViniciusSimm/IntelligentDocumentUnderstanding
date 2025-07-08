#!/bin/sh
ollama serve &
sleep 3
ollama pull qwen3:4b
wait %1