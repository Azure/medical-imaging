data "azurerm_client_config" "current" {}

resource "random_pet" "prefix" {
  prefix = var.prefix
  length = 2
}

resource "random_integer" "suffix" {
  min = 10000000
  max = 99999999
}
