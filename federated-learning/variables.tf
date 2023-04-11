# Input Variables
variable "resource_group_name" {
  type    = string
  default = "rg-medical-imaging-test-rafl"
}
variable "contact" {
  type    = string
  default = "johannes@dataroots.io"
}
variable "customer" {
  type    = string
  default = "dataroots"
}
variable "environment" {
  type        = string
  description = "Name of the environment"
  default     = "dev"
}

variable "location" {
  type        = string
  description = "Location of the resources"
  default     = "westeurope"
}

variable "prefix" {
  type        = string
  description = "Prefix of the resource name"
  default     = "ml"
}
