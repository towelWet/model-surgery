from html.parser import HTMLParser


def handle_endtag(self, tag):
    self.tags.append("/" + tag.lower())     # "/" prefix marks closing tag
