from contextlib import contextmanager


@contextmanager
def no_ssl_verify():
    import ssl
    from urllib import request
    
    try:
        request.urlopen.__kwdefaults__.update({'context': ssl.SSLContext()})
        yield
    finally:
        request.urlopen.__kwdefaults__.update({'context': None})
