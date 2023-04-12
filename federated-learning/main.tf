data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "default" {
  name     = "${var.hospital_name}-rg"
  location = var.location
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

resource "random_pet" "prefix" {
  prefix = var.prefix
  length = 2
}

resource "random_integer" "suffix" {
  min = 10000000
  max = 99999999
}
