import networkx as nx


class PageRank:
    def __init__(self, G, start_node=None, personalization=None, alpha=0.85, max_iter=100,
                 tol=1.0e-6, dangling=None, weight='weight'):
        self.G = G
        self.D = G
        self.start_node = start_node
        self.personalization = personalization
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tol
        self.dangling = dangling
        self.weight = weight

    def page_rank(self):
        if len(self.G) == 0:
            return {}

        W = nx.stochastic_graph(self.D, weight=self.weight)
        N = W.number_of_nodes()

        if self.start_node is None:
            x = dict.fromkeys(W, 1.0 / N)
        else:
            # get normalized start vector
            s = float(sum(self.start_node.values()))
            x = dict((k, v / s) for k, v in self.start_node.items())

        if self.personalization is None:

            # Assign uniform personalization vector if not given
            p = dict.fromkeys(W, 1.0 / N)
        else:
            missing = set(self.G) - set(self.personalization)
            if missing:
                raise nx.NetworkXError('Personalization dictionary ',
                                       'must have a value for every node. ',
                                       'Missing nodes %s' % missing)
            s = float(sum(self.personalization.values()))
            p = dict((k, v / s) for k, v in self.personalization.items())

        if self.dangling is None:

            # Use personalization vector if dangling vector not specified
            dangling_weights = p
        else:
            missing = set(self.G) - set(self.dangling)
            if missing:
                raise nx.NetworkXError('Dangling node dictionary ',
                                       'must have a value for every node. ',
                                       'Missing nodes %s' % missing)
            s = float(sum(self.dangling.values()))
            dangling_weights = dict((k, v / s) for k, v in self.dangling.items())
        dangling_nodes = [n for n in W if W.out_degree(n, weight=self.weight) == 0.0]

        # power iteration: make up to max_iter iterations
        for _ in range(self.max_iter):
            xlast = x
            x = dict.fromkeys(xlast.keys(), 0)
            danglesum = self.alpha * sum(xlast[n] for n in dangling_nodes)
            for n in x:

                # this matrix multiply looks odd because it is
                # doing a left multiply x^T=xlast^T*W
                for nbr in W[n]:
                    x[nbr] += self.alpha * xlast[n] * W[n][nbr][self.weight]
                x[n] += danglesum * dangling_weights[n] + (1.0 - self.alpha) * p[n]

                # check convergence, l1 norm
            err = sum([abs(x[n] - xlast[n]) for n in x])
            if err < N * self.tolerance:
                return x
        raise nx.NetworkXError('pagerank: power iteration failed to converge ',
                               'in %d iterations.' % self.max_iter)



