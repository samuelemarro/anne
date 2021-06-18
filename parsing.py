import requests

from lxml import html as htmlparse

BASE_URL ='https://www.bing.com/images/search?view=detailv2&iss=sbi&form=SBIVSP&q=imgurl:{url}&idpbck=1&idpp=skill&vt=2&sk=1&skids=SimilarImages&scode=ZHIAQZ'

def get_html(url):
    response = requests.get(url)
    return response.text

def query_bing(image_url):
    query_url = BASE_URL.format(url=image_url)
    html = get_html(query_url)
    html_tree = htmlparse.fromstring(html)

    return html_tree