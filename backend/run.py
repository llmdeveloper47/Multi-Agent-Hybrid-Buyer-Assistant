#!/usr/bin/env python3
"""
Quick start script for the Buyer Advisor API server.

Usage:
    python run.py
    python run.py --port 8080
    python run.py --no-reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the Buyer Advisor API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Hybrid Buyer Advisor - Real Estate Assistant API")
    print("=" * 60)
    print(f"\n  Starting server at http://{args.host}:{args.port}")
    print(f"  API Docs: http://localhost:{args.port}/docs")
    print(f"  Auto-reload: {'disabled' if args.no_reload else 'enabled'}")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

