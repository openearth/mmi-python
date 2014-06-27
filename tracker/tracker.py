import json
import itertools
from threading import Thread
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
import zmq
from flask import Flask, g


app = Flask(__name__)

def get_messages():
    messages = getattr(g, '_messages', None)
    if messages is None:
        messages = g._messages = []
    return messages

@app.route("/")
def hello():
    return "".join(str(x) for x in get_messages())


def server_pull(server_pull_port, messages):
    context = zmq.Context()
    # Socket to handle init data
    pull = context.socket(zmq.PULL)
    pull.bind(
        "tcp://*:{port}".format(port=server_pull_port)
    )
    for i in itertools.count():
        logger.info("waiting for %s", i)
        message = pull.recv()
        logger.info(message)
        messages.append(json.loads(message))

def main():
    server_pull_port = 5557
    with app.app_context():
        messages = get_messages()
        Thread(target=server_pull, args=(server_pull_port, messages)).start()
        app.run()
if __name__ == "__main__":
    main()
