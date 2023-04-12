# Input Variables
variable "resource_group_name" {
  type    = string
  default = "rg-medical-imaging-rafl"
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
  default     = "rafl"
}

variable "hospital_name" {
  type    = string
  default = "central_hospital_aicha"
}

variable "public_key" {
  type    = string
  default = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCyh5QGMNTxMJ1F7rr5cM4MAOACLFj5wK77EblflO1B861BkoGf9ytg/CKh6fh+Jbs3CSzTg+jkFOT3KCtBFuMrB21cXQRf10cPVbixx7MNd6SwHJC2OA7V3dDRk97Ol4ZIIDAJP9aB/uHm6N/Zqr34JqzZi/23Q9sYjKdkIZNv5Omveu5po1kN2LIjucMt2JtdcCIeRPc6tvgmvCPJhXea35xveZhDkGh9QL47/YfdsK0NjoFQs89ScvaenaB+Dbcfzb4j6Jz+IpY/3QMywlS45Ee465Ga9HDrljZ/FwuqvBLMAgEQDKBWLUbaEvtuJiydix2JDMZ3fQdW9M8twnrD3NyuZANMzvpW4eMGXdOMMFIyF5JLkdGuiQ8OT9z5ZjA6ZjXNsZgaIuNPPpdZlGey49Rm2vV30D30k8dDH3EboU1k2n5P+wN4mAgwod99XQErXvyWcPY5NCQzRi3G3Rtg2NH0CDpsKjkc3y8qcFaD+X923V5gtxazS+zVGlrk6tM= johan@Johanness-MacBook-Pro.local"

}
