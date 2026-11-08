from html.parser import HTMLParser


def initialise(self):
    HTMLParser.__init__(self)
    self.tags = []                          # "div", "/div" … in document order
    self.errors = []                        # parser complaints (counted, not raised)
