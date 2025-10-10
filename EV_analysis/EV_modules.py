import sys, polars as pl, os
if __name__ == "__main__":
    usage = lambda: print("Usage: python ModalityExtractor <input_parquet> <ekg_col> <eda_col> <trigger_col>") or sys.exit(1)
    get_output_filename = lambda base, idx: f"{base}_channel{idx+1}.parquet"
    run = lambda input_path, ekg, eda, trigger_col: (
        (lambda df:
            (lambda base:
                (
                    df.select([ekg]).write_parquet(os.path.join(f"{base}_ecgExtr", get_output_filename(base, 0))) if ekg in df.columns else None,
                    df.select([eda]).write_parquet(os.path.join(f"{base}_edaExtr", get_output_filename(base, 1))) if eda in df.columns else None,
                    df.select([trigger_col]).write_parquet(os.path.join(f"{base}_triggerExtr", get_output_filename(base, 2))) if trigger_col in df.columns else None,
                    df.select(df.columns[8:-1]).write_parquet(os.path.join(f"{base}_eegExtr", get_output_filename(base, 3))) if len(df.columns) > 9 else None
                )
            )(os.path.splitext(os.path.basename(input_path))[0])
        )(pl.read_parquet(input_path))
    )

    try:
        args = sys.argv
        (lambda a: usage() if len(a) < 5 else run(a[1], a[2], a[3], a[4]))(args)
    except Exception as e:
        print(f"[Nextflow] EV_modules errored. Error: {e}")
        sys.exit(1)