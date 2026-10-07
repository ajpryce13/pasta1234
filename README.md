# pasta1234
# Candidates: mostly topic codes, not omnibus
codes = cs["codes"].explode()                                   # one row per (code set, code)
cs["n_codes"] = codes.groupby(level=0).size()
cs["purity"] = codes.isin(set(enriched["sym"])).groupby(level=0).mean()
cap = core_codes.str.len().quantile(CODE_CAP_Q)                 # per family
cand = cs.index[(cs["purity"] >= PURITY_MIN) & (cs["n_codes"] <= cap)].to_numpy()

# C[i, k] = 1 if candidate i covers code set k (cosine >= TAU); chunks keep memory ~400 MB
step = max(1, int(1e8 // len(X)))
C = sp.vstack([sp.csr_matrix(X[cand[i:i + step]] @ X.T >= TAU, dtype=np.float32)
               for i in range(0, len(cand), step)]).tocsr()

# Greedy: add the candidate covering the most uncovered training families
test = np.random.default_rng(RANDOM_SEED).random(len(cs)) < TEST_SHARE
left = np.where(test, 0, cs["n_fam"]) / cs.loc[~test, "n_fam"].sum()   # uncovered training weight
picked = []
while True:
    gain = C @ left
    j = int(np.argmax(gain))
    if gain[j] < MIN_GAIN:
        break
    picked.append(j)
    left[C[j].indices] = 0

# Exemplars: seed = most common seed among its codes (-1 = singleton symbols only), plus a representative family
seed_of = dict(zip(seed_map["sym"], seed_map["seed"]))
rep = (core_codes.reset_index().merge(lookup, on="fam", how="left")
                 .sort_values("title", na_position="last").drop_duplicates("codes"))
exemplars = cs.loc[cand[picked]].reset_index(drop=True).merge(rep, on="codes", how="left")
exemplars["seed"] = (exemplars["codes"].explode().map(seed_of).groupby(level=0)
                     .agg(lambda s: s.mode().iat[0] if s.notna().any() else -1).astype(int))

# Coverage: each code set's best exemplar
S = X @ X[cand[picked]].T
covered = pd.DataFrame({"n_fam": cs["n_fam"], "test": test,
                        "seed": exemplars["seed"].to_numpy()[S.argmax(axis=1)]})[S.max(axis=1) >= TAU]
print(f"{len(cand):,} candidates -> {len(exemplars)} exemplars; held-out core coverage: "
      f"{covered.loc[covered.test, 'n_fam'].sum() / cs.loc[test, 'n_fam'].sum():.1%}")

per_seed = pd.DataFrame({"n_exemplars": exemplars["seed"].value_counts(),
                         "core_share": covered.groupby("seed")["n_fam"].sum() / covered["n_fam"].sum()}
                        ).fillna(0).sort_values("core_share", ascending=False)
display(per_seed.round(3))
exemplars[["seed", "publication_number", "title", "n_codes", "purity"]].head(30)
