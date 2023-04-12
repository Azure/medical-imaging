# Configure the AzureRM Provider
provider "azurerm" {
  features {
  }
}

# Dependent resources for Azure Machine Learning
resource "azurerm_application_insights" "default" {
  count               = 3
  name                = "${random_pet.prefix.id}-appi${count.index}"
  location            = azurerm_resource_group.default.location
  resource_group_name = azurerm_resource_group.default.name
  application_type    = "web"
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

resource "azurerm_key_vault" "default" {
  count                    = 3
  name                     = "${var.prefix}${var.environment}${random_integer.suffix.result}kv${count.index}"
  location                 = azurerm_resource_group.default.location
  resource_group_name      = azurerm_resource_group.default.name
  tenant_id                = data.azurerm_client_config.current.tenant_id
  sku_name                 = "premium"
  purge_protection_enabled = false
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

resource "azurerm_storage_account" "default" {
  count                           = 3
  name                            = "${var.prefix}${var.environment}${random_integer.suffix.result}st${count.index}"
  location                        = azurerm_resource_group.default.location
  resource_group_name             = azurerm_resource_group.default.name
  account_tier                    = "Standard"
  account_replication_type        = "GRS"
  allow_nested_items_to_be_public = false
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

resource "azurerm_container_registry" "default" {
  count               = 3
  name                = "${var.prefix}${var.environment}${random_integer.suffix.result}cr${count.index}"
  location            = azurerm_resource_group.default.location
  resource_group_name = azurerm_resource_group.default.name
  sku                 = "Premium"
  admin_enabled       = true
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

# Machine Learning workspace
resource "azurerm_machine_learning_workspace" "default" {
  count                         = 3
  name                          = "${random_pet.prefix.id}-mlw${count.index}"
  location                      = azurerm_resource_group.default.location
  resource_group_name           = azurerm_resource_group.default.name
  application_insights_id       = azurerm_application_insights.default[count.index].id
  key_vault_id                  = azurerm_key_vault.default[count.index].id
  storage_account_id            = azurerm_storage_account.default[count.index].id
  container_registry_id         = azurerm_container_registry.default[count.index].id
  public_network_access_enabled = true
  tags = {
    contact  = var.contact
    customer = var.customer
  }

  identity {
    type = "SystemAssigned"
  }
}
