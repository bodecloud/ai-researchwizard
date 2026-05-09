from itertools import islice
from ..utils import check_pkg


class Duckduckgo:
    """
    Duckduckgo API Retriever
    """
    def __init__(self, query, query_domains=None):
        check_pkg('ddgs')
        from ddgs import DDGS
        self.ddg = DDGS()
        self.query = query
        self.query_domains = query_domains or None

    def _scoped_query(self):
        if not self.query_domains:
            return self.query
        domains = [str(domain).strip().removeprefix('www.') for domain in self.query_domains if str(domain).strip()]
        if not domains:
            return self.query
        site_clause = ' OR '.join(f'site:{domain}' for domain in domains)
        return f'{self.query} ({site_clause})'

    def search(self, max_results=5):
        """
        Performs the search
        :param query:
        :param max_results:
        :return:
        """
        try:
            search_response = self.ddg.text(self._scoped_query(), region='wt-wt', max_results=max_results)
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []
        return search_response
