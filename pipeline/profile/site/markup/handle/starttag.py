from html.parser import HTMLParser


def handle_starttag(self, tag, attrs):
    self.tags.append(tag.lower())
