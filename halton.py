import jax
import jax.numpy as np
import matplotlib.pyplot as plt

def halton_sequence(n, rng):
    return randradinv(np.arange(n), 2, rng)

def randradinv(ind, b, rng):
    br = 1 / b
    ans = ind * 0
    res = ind
    while(1 - br < 1):
        digit = np.fmod(res, b).astype(int)
        rng, key = jax.random.split(rng)
        permutation = jax.random.permutation(key, b)
        perm_digit = permutation[digit]
        ans += perm_digit * br
        br /= b
        res = (res - digit) / b
    return ans

if __name__ == "__main__":
    rng = jax.random.PRNGKey(3276428)
    seq = halton_sequence(100, rng)
    print(seq)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(seq)
    ax[1].plot(jax.random.uniform(jax.random.PRNGKey(48279), (100,)))
    plt.show()
