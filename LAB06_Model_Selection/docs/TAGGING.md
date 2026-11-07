# Historical Week 7 / 8 Git tags — pending

The original experiment is preserved on branch `lab06-model-selection-2473986-20261025`. The current GitHub connection supports branch updates but does **not** expose creation of refs under `refs/tags`. **No git tags were created by this patch.**

After checking actual historical commits, an authenticated local Git client may create and push tags:

```bash
git fetch origin lab06-model-selection-2473986-20261025
git log --oneline -- LAB06_Model_Selection
git tag -a lab06-week7-verified <actual-week7-commit-sha> -m "Lab 06 Week 7"
git tag -a lab06-week8-verified <actual-week8-commit-sha> -m "Lab 06 Week 8"
git push origin lab06-week7-verified lab06-week8-verified
```

Do not guess a historical commit SHA and do not pretend that today's documentation fixes existed before the experiment.
