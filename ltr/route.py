import argparse
import json

from .embed import Embedder
from .router import Router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--router", default="experiments/router.json")
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    router = Router.load(args.router, embedder=Embedder(args.backend))
    print(json.dumps(router.route(args.text, quality_floor=args.floor), indent=2))


if __name__ == "__main__":
    main()
