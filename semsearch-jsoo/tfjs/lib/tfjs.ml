open Js_of_ocaml
include Tfjs_intf

type model
type io_handler

module Tensor = struct
  type t

  let to_array tensor : float Array.t =
    let output =
      Js.Unsafe.meth_call (Js.Unsafe.inject tensor) "dataSync" [||]
    in
    Js.to_array output
end

let require pkg =
  lazy
    (Fmt.pr "JS require('%s')@." pkg;
     Js.Unsafe.fun_call
       (Js.Unsafe.js_expr "require")
       [| Js.string pkg |> Js.Unsafe.inject |])

let tf = require "@tensorflow/tfjs"
let tfn = require "@tensorflow/tfjs-node"

(** This is required to register Node-specific backends. *)

let load_graph_model handler =
  Js.Unsafe.fun_call
    (Lazy.force tf)##.loadGraphModel
    [| Js.Unsafe.inject handler |]
  |> Promise.to_lwt

let load_saved_model path =
  Js.Unsafe.fun_call
    (Lazy.force tfn)##.node##.loadSavedModel
    [| Js.string path |> Js.Unsafe.inject |]
  |> Promise.to_lwt

let ready () = Js.Unsafe.fun_call (Lazy.force tf)##.ready [||] |> Promise.to_lwt

let tensor2d data shape dtype =
  Js.Unsafe.fun_call
    (Lazy.force tf)##.tensor2d
    [|
      Js.Unsafe.inject data;
      Js.array shape |> Js.Unsafe.inject;
      Dtype.to_string dtype |> Js.string |> Js.Unsafe.inject;
    |]

let zeros shape dtype =
  Js.Unsafe.fun_call
    (Lazy.force tf)##.zeros
    [|
      Js.array shape |> Js.Unsafe.inject;
      Dtype.to_string dtype |> Js.string |> Js.Unsafe.inject;
    |]

let matmul tensor1 tensor2 =
  Js.Unsafe.fun_call
    (Lazy.force tf)##.matMul
    [| Js.Unsafe.inject tensor1; Js.Unsafe.inject tensor2 |]

let transpose tensor =
  Js.Unsafe.fun_call (Lazy.force tf)##.transpose [| Js.Unsafe.inject tensor |]

let reshape tensor (shape : int array) =
  let shape_js = Js.array shape in
  Js.Unsafe.meth_call tensor "reshape" [| Js.Unsafe.inject shape_js |]

module IO = struct
  (* Custom IOHandler that does not require tfjs-node. *)
  let local_fs_handler path =
    let constr = Js.Unsafe.js_expr "LocalFSHandler" in
    new%js constr (path |> Js.string |> Js.Unsafe.inject)

  let fileSystem path =
    Js.Unsafe.fun_call
      (Lazy.force tfn)##.io##.fileSystem
      [| path |> Js.string |> Js.Unsafe.inject |]
end

module Backend = struct
  type t =
    | Pure_js  (** You will die of old age before this finishes running *)
    | Wasm
        (** Low-latency and easily portable across Node.js, browser, Electron *)
    | Native_tensorflow  (** C++ *)
    | Webgpu
        (** Meant to be the highest-performing backend, and should work under Node *)
    | Webgl  (** Use the GPU on older browsers *)
    | Other of string (* React... *)

  let to_string = function
    | Pure_js -> "cpu" (* Yes, I know. *)
    | Native_tensorflow -> "node"
    | Webgpu -> "webgpu"
    | Webgl -> "webgl"
    | Wasm -> "wasm"
    | Other s -> s

  let of_string = function
    | "cpu" -> Pure_js
    | "node" -> Native_tensorflow
    | "webgpu" -> Webgpu
    | "webgl" -> Webgl
    | "wasm" -> Wasm
    | s -> Other s

  let wasm_backend = require "@tensorflow/tfjs-backend-wasm"

  let set backend : unit Lwt.t =
    (* XXX: it's a more complicated story for some other backends. *)
    (match backend with
    | Native_tensorflow -> ignore (Lazy.force tfn)
    | Wasm -> ignore (Lazy.force wasm_backend)
    | _ -> ());

    let backend = to_string backend in

    Js.Unsafe.fun_call
      (Lazy.force tf)##.setBackend
      [| Js.string backend |> Js.Unsafe.inject |]
    |> Promise.to_lwt

  let get () : string =
    Js.Unsafe.fun_call (Lazy.force tf)##.getBackend [||] |> Js.to_string
end
