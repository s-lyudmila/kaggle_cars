def prime(n):
    if n <= 1:
        return False
    if n >= 2:
        for i in range(2, n):
            if not (n % i):
                return False
        return True


def prime_maxlength_chain(n):
    primes_list = [i for i in range(n) if prime(i)]
    sums = [primes_list[j:i] for i in range(n) for j in range(n) if (sum(primes_list[j:i]) <= n) and (sum(primes_list[j:i]) > 0)]
    for i in sorted(sums):
        print(sum(i))





n = 499
prime_maxlength_chain(n)