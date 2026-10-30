# Synthetic datasets

`synthetic_datasets.zip` contains the 250 synthetic instances used in the article:
50 instances for each of the groups Dispersion-Low, BASE, Dispersion-High,
Offer-Low and Offer-High (`Synthetic/<group>/instance_NNN.json`, legacy schema).
They were produced with the repository generator and default parameters:

```powershell
.venv\Scripts\python.exe -m uctp.generation --output datasets/Synthetic/generated --groups Dispersion-Low BASE Dispersion-High Offer-Low Offer-High --count 50 --seed 1
```

Running the command again reproduces the files byte for byte; `SHA256SUMS`
lists their checksums (paths relative to `Synthetic/`) and `generation_log.txt`
the console output, including the candidate seed of every saved instance and
the number of candidates tested per group.

Read the files with `--input-format legacy`. Within one instance number, the
Dispersion-Low, BASE and Dispersion-High instances share the candidate seed and
therefore the course, section, teacher and availability structure; they differ
only in the curricular sampling of student demand.
