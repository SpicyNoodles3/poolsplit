# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-07

### Added

- Initial release.
- `Pool` dataclass describing a reserved per-type token/entry budget, with
  validation of non-negative limits.
- `retrieve` — fill one pool: type/active filtering, always-scope entries loaded
  unconditionally outside budget competition, deterministic scored ordering, and
  greedy budget fill with a never-zero cost rule for entries missing a
  `token_count`.
- `retrieve_pools` — fill several pools in priority order with identity-based
  dedupe across pools and duplicate-name rejection.
- `default_scorer` — lexical scorer (base priority × recency × tag overlap ×
  project match × message overlap), with an injectable `now` and no
  access-frequency term.
- `default_token_counter` and `default_is_active` defaults, both pluggable.
