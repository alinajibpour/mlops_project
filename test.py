import pstats

p=pstats.Stats("vae_mnist_working.prof")
p.strip_dirs().sort_stats("cumtime").print_stats(10)
