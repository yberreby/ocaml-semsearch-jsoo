let sbert_model = "all-MiniLM-L6-v2"
(* let sbert_model = "paraphrase-MiniLM-L3-v2" *)

let test_root = Sys.getcwd ()

(* FIXME: there has to be a better way than this horror *)
(* Using Dune dependencies and source_tree doesn't work on parent directories, though. *)
let root = test_root ^ "../../../../../"

(* TODO: proper path handling *)
let vocab = root ^ "node_modules/bert-tokenizer/assets/vocab.json"
let exported_path = root ^ "exported/"
let tfjs_path = exported_path ^ "js/" ^ "sentence-transformers-" ^ sbert_model

(* Load list of lines, stripped of whitespace *)
let sentences : string List.t =
  let lines = ref [] in
  let chan = open_in (test_root ^ "test_tasks.txt") in
  try
    while true do
      let line = input_line chan in
      lines := String.trim line :: !lines
    done;
    []
  with End_of_file ->
    close_in chan;
    List.rev !lines

let top_n n lst =
  let len = List.length lst in
  if len < n then lst else List.filteri (fun i _ -> i < n) lst

let test_queries =
  [
    "schoolwork";
    "work";
    "paperwork";
    "languages";
    "house";
    "duck";
    "Python";
    "stressful";
    "activities good for my mental health";
    "war";
    "CEO";
  ]

let cur_s = Unix.gettimeofday

let f () =
  let backend = Tfjs.Backend.Wasm in
  let%lwt () = Tfjs.Backend.set backend in
  let start = cur_s () in
  let handler = tfjs_path |> Tfjs.IO.local_fs_handler in
  let%lwt model = Tfjs.load_graph_model handler in
  Fmt.pr "Model loaded in %.3fs\n@." (cur_s () -. start);
  let tokenizer = Sbert_jsoo.SBERT.Tokenizer.with_vocab vocab in
  let start = cur_s () in
  let encoded_sentences = Sbert_jsoo.SBERT.encode model tokenizer sentences in
  Fmt.pr "Encoded %d sentences in %.3fs\n" (List.length sentences)
    (cur_s () -. start);
  let find_with_query base_sentence =
    let start = cur_s () in
    let base_encoded =
      Sbert_jsoo.SBERT.encode model tokenizer [ base_sentence ]
    in
    Fmt.pr "Encoded \"%s\" in %.3fs\n" base_sentence (cur_s () -. start);
    let start_search = cur_s () in
    let similarities =
      Semsearch_jsoo.calculate_similarities_with_base base_encoded
        encoded_sentences
    in
    let end_search = cur_s () in
    Fmt.pr " Top matches (found in %.3fs):\n" (end_search -. start_search);
    let sentences = Array.of_list sentences in
    List.iter
      (fun (i, similarity) ->
        Fmt.pr "  '%s' with similarity %.2f\n" sentences.(i) similarity)
      (top_n 3 (similarities |> Array.to_list));
    Fmt.pr "@."
  in
  Base.List.iter test_queries ~f:find_with_query;
  Lwt.return_unit

let () =
  Fmt.pr "Running under JSOO@.";
  Lwt.async f;
  ()
