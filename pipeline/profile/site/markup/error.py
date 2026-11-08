from html.parser import HTMLParser


def error(self, message):
    self.errors.append(message)
