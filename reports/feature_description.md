# Feature Descriptions

This document describes all features in the supply chain dataset.

## Feature Catalog

| Feature Name | Description | Data Type | Notes |
|-------------|-------------|-----------|-------|
| **Type** | Type of transaction made | Categorical | Transaction classification |
| **Days for shipping (real)** | Actual shipping days of the purchased product | Numeric | Real delivery time |
| **Days for shipment (scheduled)** | Days of scheduled delivery of the purchased product | Numeric | Planned delivery time |
| **Benefit per order** | Earnings per order placed | Numeric | Profit metric |
| **Sales per customer** | Total sales per customer made per customer | Numeric | Customer-level sales aggregate |
| **Delivery Status** | Delivery status of orders: Advance shipping, Late delivery, Shipping canceled, Shipping on time | Categorical | Delivery outcome status |
| **Late_delivery_risk** | Categorical variable that indicates if sending is late (1), it is not late (0) | Binary | 1 = late, 0 = on time |
| **Category Id** | Product category code | Numeric/ID | Category identifier |
| **Category Name** | Description of the product category | Categorical | Category label |
| **Customer City** | City where the customer made the purchase | Categorical | Customer location - city |
| **Customer Country** | Country where the customer made the purchase | Categorical | Customer location - country |
| **Customer Email** | Customer's email | String | Customer contact information |
| **Customer Fname** | Customer name | String | Customer first name |
| **Customer Id** | Customer ID | Numeric/ID | Unique customer identifier |
| **Customer Lname** | Customer lastname | String | Customer last name |
| **Customer Password** | Masked customer key | String | Encrypted/masked authentication |
| **Customer Segment** | Types of Customers: Consumer, Corporate, Home Office | Categorical | Customer classification |
| **Customer State** | State to which the store where the purchase is registered belongs | Categorical | Customer location - state |
| **Customer Street** | Street to which the store where the purchase is registered belongs | String | Customer location - street address |
| **Customer Zipcode** | Customer Zipcode | String/Numeric | Customer location - postal code |
| **Department Id** | Department code of store | Numeric/ID | Department identifier |
| **Department Name** | Department name of store | Categorical | Department label |
| **Latitude** | Latitude corresponding to location of store | Numeric | Geographic coordinate |
| **Longitude** | Longitude corresponding to location of store | Numeric | Geographic coordinate |
| **Market** | Market to where the order is delivered: Africa, Europe, LATAM, Pacific Asia, USCA | Categorical | Regional market classification |
| **Order City** | Destination city of the order | Categorical | Order destination - city |
| **Order Country** | Destination country of the order | Categorical | Order destination - country |
| **Order Customer Id** | Customer order code | Numeric/ID | Order-customer relationship identifier |
| **order date (DateOrders)** | Date on which the order is made | DateTime | Order timestamp |
| **Order Id** | Order code | Numeric/ID | Unique order identifier |
| **Order Item Cardprod Id** | Product code generated through the RFID reader | Numeric/ID | RFID product identifier |
| **Order Item Discount** | Order item discount value | Numeric | Discount amount |
| **Order Item Discount Rate** | Order item discount percentage | Numeric | Discount percentage (0-100) |
| **Order Item Id** | Order item code | Numeric/ID | Unique order item identifier |
| **Order Item Product Price** | Price of products without discount | Numeric | Base product price |
| **Order Item Profit Ratio** | Order Item Profit Ratio | Numeric | Profitability metric |
| **Order Item Quantity** | Number of products per order | Numeric | Quantity ordered |
| **Sales** | Value in sales | Numeric | Sales amount |
| **Order Item Total** | Total amount per order | Numeric | Total order value |
| **Order Profit Per Order** | Order Profit Per Order | Numeric | Profit per order |
| **Order Region** | Region of the world where the order is delivered: Southeast Asia, South Asia, Oceania, Eastern Europe, etc. | Categorical | Order destination - region |
| **Order State** | State of the region where the order is delivered | Categorical | Order destination - state |
| **Order Status** | Order Status: COMPLETE, PENDING, CLOSED, PENDING_PAYMENT, CANCELED, PROCESSING, SUSPECTED_FRAUD | Categorical | Order lifecycle status |
| **Product Card Id** | Product code | Numeric/ID | Product identifier |
| **Product Category Id** | Product category code | Numeric/ID | Product category identifier |
| **Product Description** | Product Description | String | Product details |
| **Product Image** | Link of visit and purchase of the product | String/URL | Product image URL |
| **Product Name** | Product Name | String | Product title |
| **Product Price** | Product Price | Numeric | Product unit price |
| **Product Status** | Status of the product stock: If it is 1 not available, 0 the product is available | Binary | 1 = out of stock, 0 = available |
| **Shipping date (DateOrders)** | Exact date and time of shipment | DateTime | Shipment timestamp |
| **Shipping Mode** | The following shipping modes are presented: Standard Class, First Class, Second Class, Same Day | Categorical | Shipping service level |

## Feature Categories

### Customer Features
- Customer Id, Customer Fname, Customer Lname, Customer Email, Customer Password
- Customer City, Customer Country, Customer State, Customer Street, Customer Zipcode
- Customer Segment

### Order Features
- Order Id, Order Customer Id, order date (DateOrders)
- Order City, Order Country, Order State, Order Region, Order Status
- Order Item Id, Order Item Quantity, Order Item Product Price, Order Item Discount, Order Item Discount Rate
- Order Item Total, Order Item Profit Ratio, Order Profit Per Order
- Sales, Benefit per order

### Product Features
- Product Card Id, Product Name, Product Description, Product Image
- Product Category Id, Product Price, Product Status
- Category Id, Category Name

### Shipping Features
- Shipping Mode, Shipping date (DateOrders)
- Days for shipping (real), Days for shipment (scheduled)
- Delivery Status, Late_delivery_risk

### Location Features
- Latitude, Longitude
- Market

### Store Features
- Department Id, Department Name
