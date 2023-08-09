open Base
open Js_of_ocaml

module SBERT : sig
  module Tokenizer : sig
    type t

    val with_vocab : string -> t
    val tokenize : t -> string -> int Array.t
  end

  val encode : Tfjs.model -> Tokenizer.t -> string list -> Tfjs.Tensor.t
end = struct
  let start_token_id = 101
  let end_token_id = 102
  let embedding_dim = 384

  module Tokenizer = struct
    type t = Js.Unsafe.any

    let bert_tokenizer =
      lazy (Js.Unsafe.js_expr "require('bert-tokenizer').BertTokenizer")

    let with_vocab vocabUrl =
      let constr = Lazy.force bert_tokenizer in
      new%js constr
        (Js.Unsafe.inject vocabUrl)
        (Js.Unsafe.inject (Js.bool true))

    let tokenize tokenizer input : int Array.t =
      Js.Unsafe.meth_call tokenizer "tokenize"
        [| Js.Unsafe.inject (Js.string input) |]
      |> Js.to_array
  end

  let create_tensors (tokens_batch : int Array.t Array.t) =
    let batch_size = Array.length tokens_batch in
    let max_sequence_length = Array.get tokens_batch 0 |> Array.length in
    let shape = [| batch_size; max_sequence_length |] in
    let input_ids =
      Tfjs.tensor2d
        (Array.map ~f:Js.array tokens_batch |> Js.array)
        shape Tfjs.Dtype.Int32
    in
    let token_type_ids = Tfjs.zeros shape Tfjs.Dtype.Int32 in
    let attention_mask =
      Array.map tokens_batch ~f:(fun tokens ->
          Array.map ~f:(fun token -> if token = 0 then 0 else 1) tokens)
    in
    let attention_mask =
      Tfjs.tensor2d
        (Array.map ~f:Js.array attention_mask |> Js.array)
        [| batch_size; max_sequence_length |]
        Tfjs.Dtype.Int32
    in
    (input_ids, token_type_ids, attention_mask)

  let pad_tokens ~max_length (tokens : int Array.t) =
    let l : int = Array.length tokens in
    let pad_size = max_length - l in
    if pad_size < 0 then failwith "Input sequence too long"
    else
      let pad = Stdlib.Array.make pad_size 0 in
      Array.concat
        [
          Stdlib.Array.make 1 start_token_id;
          tokens;
          Stdlib.Array.make 1 end_token_id;
          pad;
        ]

  let encode (model : Tfjs.model) tokenizer sentences =
    (* Tokenize *)
    let tokens_batch =
      List.map sentences ~f:(Tokenizer.tokenize tokenizer) |> Array.of_list
    in
    (* Pad to longest sequence *)
    let lengths = Array.map tokens_batch ~f:Array.length in
    let max_length =
      lengths |> Array.max_elt ~compare:Int.compare |> Option.value_exn
    in
    let tokens_batch = Array.map tokens_batch ~f:(pad_tokens ~max_length) in
    (* Create tensors *)
    let input_ids, token_type_ids, attention_mask =
      create_tensors tokens_batch
    in
    let inputs =
      Js.Unsafe.obj
        [|
          ("input_ids", Js.Unsafe.inject input_ids);
          ("token_type_ids", Js.Unsafe.inject token_type_ids);
          ("attention_mask", Js.Unsafe.inject attention_mask);
        |]
    in
    (* Predict *)
    let output =
      Js.Unsafe.meth_call (Js.Unsafe.inject model) "predict"
        [| Js.Unsafe.inject inputs |]
    in
    Tfjs.reshape output [| -1; embedding_dim |]
end
