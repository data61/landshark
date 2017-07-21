import logging
import click
import tensorflow as tf
import signal
import sys

def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

log = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(verbosity):
    logging.basicConfig()
    lg = logging.getLogger("")
    lg.setLevel(verbosity)


@cli.command()
@click.argument("task_number", type=int)
def run(task_number):
    cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
    server = tf.train.Server(cluster, job_name="local", task_index=task_number)
    log.info("Starting server #{}".format(task_number))
    server.start()
    # if task_namber == 0:
    #     signal.pause()
    log.info("Started server #{}".format(task_number))
    x = tf.constant(2)


    with tf.device("/job:local/task:1"):
        y2 = x - 66

    with tf.device("/job:local/task:0"):
        y1 = x + 300
        y = y1 + y2


    with tf.Session("grpc://localhost:2222") as sess:
        result = sess.run(y)
        print(result)


