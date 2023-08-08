open Js_of_ocaml
include module type of Tfjs_intf

type model
type io_handler

module Tensor : sig
  type t

  val to_array : t -> float Array.t
end

val load_graph_model : io_handler -> model Lwt.t

val tensor2d : 'a Js.t -> int Array.t -> Dtype.t -> Tensor.t
(** Data can be a flat or nested Array, or a TypedArray. *)

val zeros : int Array.t -> Dtype.t -> Tensor.t
val matmul : Tensor.t -> Tensor.t -> Tensor.t
val transpose : Tensor.t -> Tensor.t
val reshape : Tensor.t -> int array -> Tensor.t

module IO : sig
  val fileSystem : string -> io_handler
end
