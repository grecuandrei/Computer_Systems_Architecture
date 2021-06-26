import json
import matplotlib.pyplot as plt

RESULTS_FILE='ibm_nehalem_runtimes.json'

with open(RESULTS_FILE) as runtimes_file:
	methods = json.load(runtimes_file)

for method in methods['methods']:
	plt.plot(methods['N'], method['runtimes'], label=method['name'])

plt.ylabel('Elapsed Time (s)')
plt.xlabel('N = Size of Matrix')
plt.title('Performance of the Implementations')
plt.ylim(top=methods['y_top'])
plt.legend()

plt.savefig('graphic.png')

plt.ylim(top=methods['y_top_extra'])
plt.legend()

plt.savefig('graphic_extra.png')
