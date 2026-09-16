bash setup_env.sh
uv run download.py
LTX_MULTI_MODE=16 LTX_HOST="0.0.0.0" LTX_API_TOKEN="lotusmind" uv run python ltx_server.py
