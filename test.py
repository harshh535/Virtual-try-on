 def get_opt():
     parser = argparse.ArgumentParser(description="Test Virtual Try-On")
     # ─── Existing arguments ───────────────────────────────────────────────────
     parser.add_argument(
         "--init_type",
         type=str,
         default="normal",
         help="network weight initialization (default: normal)"
     )
     parser.add_argument(
         "--init_variance",
         type=float,
         default=0.02,
         help="initialization variance (default: 0.02)"
     )
     parser.add_argument("--name",            type=str, required=True)
     parser.add_argument("--dataset_dir",     type=str, default="./datasets")
     parser.add_argument(
         "--dataset_mode",
         type=str,
         default="test",
         help="which subfolder under dataset_dir to use"
     )
     parser.add_argument(
         "--dataset_list",
         type=str,
         default="test/test_pairs.txt",
         help="relative path (inside dataset_dir) listing cloth-model pairs"
     )
     parser.add_argument("--save_dir",        type=str, default="./results")
     parser.add_argument("--checkpoint_dir",  type=str, default="./checkpoints")
     parser.add_argument("--load_height",     type=int, default=1024)
     parser.add_argument("--load_width",      type=int, default=768)
     parser.add_argument("--semantic_nc",     type=int, default=13)
     parser.add_argument("--grid_size",       type=int, default=5)
     parser.add_argument("--norm_G",          type=str, default="spectralaliasinstance")
     parser.add_argument("--ngf",             type=int, default=64)
     parser.add_argument(
         "--num_upsampling_layers",
         choices=["normal", "more", "most"],
         default="most",
         help="how many upsampling layers in the network"
     )
     parser.add_argument("--display_freq",    type=int, default=1)
     # ──────────────────────────────────────────────────────────────────────────

     # ─── Add these missing arguments ──────────────────────────────────────────
     parser.add_argument(
         "--shuffle",
         action="store_true",
         help="whether to shuffle the dataset (default: False)"
     )
     parser.add_argument(
         "--batch_size",
         type=int,
         default=1,
         help="batch size for DataLoader (default: 1)"
     )
     parser.add_argument(
         "--num_workers",
         type=int,
         default=4,
         help="number of worker threads for DataLoader (default: 4)"
     )
+    parser.add_argument(
+        "--workers",
+        type=int,
+        default=4,
+        help="alias for --num_workers; used by VITONDataLoader"
+    )
     # ──────────────────────────────────────────────────────────────────────────

     return parser.parse_args()
