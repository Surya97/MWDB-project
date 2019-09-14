def ranking(ar):
    n = len(ar)
    rank_vector = [0]*n
    for i in range(n):
        l = 1
        m = 1
        for j in range(i):
            if ar[j] < ar[i]:
                l += 1
            if ar[j] == ar[i]:
                m += 1

        for j in range(i+1, n):
            if ar[j] < ar[i]:
                l += 1
            if ar[j] == ar[i]:
                m += 1

        rank_vector[i] = l + (m-1) * 0.5

    return rank_vector


def correlation_coefficient(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    # sum_xy = sum([a*b for a,b in zip(x,y)])
    # sum_square_x = sum([a*a for a in x])
    # sum_square_y = sum([a*a for a in y])
    #
    # correlation = (n * sum_xy - sum_x * sum_y) / pow((n * sum_square_x - sum_x * sum_x) *
    #                                                  (n * sum_square_y - sum_y * sum_y), 0.5)

    diff_array = [a-b for a, b in zip(x, y)]

    coefficient = 1 - (sum([a*a for a in diff_array]))/(n * (n**2 - 1))

    return coefficient

