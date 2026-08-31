# slopsquatting-replication-2026

Replication of Spracklen et al. (USENIX Security '25) on the 2026 frontier-LLM cohort.

## Paper
"The Range Shrinks, the Threat Remains: Re-evaluating LLM Package
Hallucinations on the 2026 Frontier-Model Cohort"

- arXiv: https://arxiv.org/abs/2605.17062
- Zenodo: 10.5281/zenodo.19859120

## The universal-hallucination set
The paper reports a 127-name universal-hallucination set (109 PyPI, 18 npm),
of which 53 (41 PyPI, 12 npm) remain registrable after each registry's
existing defenses. Coordinated disclosure to PyPI Security and Socket.dev
completed in April–May 2026.

The full enumerated list is no longer embargoed. An earlier revision of this
repository included it in a `disclosure/` directory that was publicly
accessible from April 30, 2026; that directory has since been removed from
the current tree, but the list remains reachable in the tagged release
`v0.2-preprint`. The set is therefore treated as disclosed as of April 30,
2026.

Two caveats apply to the enumerated names, both discussed in §9.2 of the paper:

- **Screen against the registrable 53, not the raw 127.** 68 of the 109 PyPI
  names are already blocked by PyPI's prohibited-name list and
  ultranormalization; the four `@ember/*` npm names are virtual modules under
  a controlled scope, `ssh-keys` is a published package, and `metro-evaluator`
  is a security-hold placeholder.
- **Some names in the raw set are real packages.** Validation used the master
  lists shipped with the Spracklen 2025 artifact rather than a current
  registry snapshot, so names registered in the interval were counted as
  absent. A revalidation against a current snapshot is in progress and will be
  reflected in a subsequent paper revision.

## Contact
Aleksandr Churilov — churik0509@gmail.com
