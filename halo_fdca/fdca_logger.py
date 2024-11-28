import os, sys, logging, time

class _ColorStreamHandler(logging.StreamHandler):
    DEFAULT = '\x1b[0m'
    RED     = '\x1b[31m'
    GREEN   = '\x1b[32m'
    YELLOW  = '\x1b[33m'
    CYAN    = '\x1b[36m'

    CRITICAL = RED
    ERROR    = RED
    WARNING  = YELLOW
    INFO     = GREEN
    DEBUG    = CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        color = self._get_color(record.levelno)
        record.msg = color + record.msg + self.DEFAULT
        return logging.StreamHandler.format(self, record)

class Logger():
    def __init__(self, pipename, path = ""):
        logger = logging.getLogger()
        logger.propagate = False
        logger.handlers = []
        
        output_path = "" if path == "" else path + "/"
 
        timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        self.logfile = f"{output_path}{pipename}_{timestamp}.log"
        self.set_logger(self.logfile)
        os.symlink(self.logfile, pipename+'.log.tmp')
        os.rename(pipename+'.log.tmp', pipename+'.log')      

    def set_logger(self, logfile):
        logger = logging.getLogger("FDCA")
        logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        handlerFile = logging.FileHandler(logfile)
        handlerFile.setLevel(logging.DEBUG)
        
        # create console handler with a higher log level
        handlerConsole = _ColorStreamHandler(stream=sys.stdout)
        handlerConsole.setLevel(logging.INFO)
        
        # create formatter and add it to the handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        handlerFile.setFormatter(formatter)
        handlerConsole.setFormatter(formatter)
        
        # add the handlers to the logger
        logger.addHandler(handlerFile)
        logger.addHandler(handlerConsole)

        logger.info('Logging initialised in %s (file: %s)' % (os.getcwd(), logfile))


# this is used by all libraries for logging
logger = logging.getLogger("FDCA")
