let test_root = Sys.getcwd ()
(* FIXME: there has to be a better way than this horror *)
let root = test_root ^ "../../../../../"

(* TODO: proper path handling *)
let vocabUrl = root ^ "node_modules/bert-tokenizer/assets/vocab.json"

let json_path =
  root ^ "exported/js/" ^ "sentence-transformers-all-MiniLM-L6-v2/"
  ^ "model.json"

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
    "uni";
    "work";
    "paperwork";
    "languages";
    "house";
    "duck";
    "Python";
    "stressful";
    "activities good for my mental health";
  ]

let f () =
  let cur_s = Unix.gettimeofday in
  let start = cur_s () in
  let%lwt model = Semsearch_jsoo.load_local_model json_path in
  Fmt.pr "Model loaded in %.3fs\n@." (cur_s () -. start);
  let tokenizer = Sbert_jsoo.SBERT.Tokenizer.with_vocab vocabUrl in

  let start = cur_s () in
  let encoded_sentences = Sbert_jsoo.SBERT.encode model tokenizer sentences in
  Fmt.pr "Encoded %d sentences in %.3fs\n"
    (List.length sentences)
    (cur_s () -. start);

  let find_with_query base_sentence =
    let start = cur_s () in
    let base_encoded =
      Sbert_jsoo.SBERT.encode model tokenizer [ base_sentence ]
    in
    Fmt.pr "Encoded \"%s\" in %.3fs\n" base_sentence
      (cur_s () -. start);
    Fmt.pr "\nQuery: %s; top matches:\n" base_sentence;
    let start_search = cur_s () in
    let similarities =
      Semsearch_jsoo.calculate_similarities_with_base base_encoded
        encoded_sentences
    in
    let sentences = Array.of_list sentences in
    List.iter
      (fun (i, similarity) ->
        Fmt.pr " '%s' with similarity %.2f\n" sentences.(i) similarity)
      (top_n 3 similarities);
    let end_search = cur_s () in
    Fmt.pr "Searched in %.3fs\n"
      (end_search -. start_search)
  in

  Base.List.iter test_queries ~f:find_with_query;
  Lwt.return_unit

let () =
  Fmt.pr "Running under JSOO\n";
  Lwt.async f;
  ()