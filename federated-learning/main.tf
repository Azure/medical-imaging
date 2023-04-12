data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "default" {
  name     = var.resource_group_name
  location = var.location
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

resource "random_integer" "suffix" {
  min = 10000000
  max = 99999999
}
