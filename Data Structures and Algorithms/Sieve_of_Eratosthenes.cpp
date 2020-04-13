// Sieve of Eratosthenes till 10^6
void sieve(vector<int> &primes, int length = 1000001) {
    char base[length];
    memset(base, 1, sizeof(char) * (length));
    base[0] = base[1] = 0;
    // eliminate multiples of 2
    for (int i = 4; i < length; i += 2)
        base[i] = 0;
    //eliminate multiples of primes
    for (int i = 3; i < length / 2 + 1; i += 2)
        if (base[i])
            for (int j = i * 2; j < length; j += i)
                base[j] = 0;
    primes.push_back(2);
    for (int i = 3; i < length; i += 2)
        if (base[i])
            primes.push_back(i);
}