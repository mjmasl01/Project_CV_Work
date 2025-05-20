

library(shiny)
library(tidyverse)
library(plotly)

# Reading the data
alc <- read_csv("Alcohol_Consumption_US.csv")

# List of states
states <- c("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
            "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
            "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
            "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
            "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
            "Wisconsin", "Wyoming")

# Define UI for application
ui <- fluidPage(
  tags$style(HTML("body {background-color: lightblue; }")),
  titlePanel(tags$h1("US Alcohol Consumption (Per Capita)", style = "font-weight: bold; text-align: center; color: darkblue; font-size: 40px;")),
  tags$p("Author: Matt Maslow", style = "font-size: 21px;  text-align: center; color: black;"),
  # print a message for the user
  tags$p("For the States option.... if you select 'All' then will see a BarPlot.... Otherwise, selecting any other 'State' will produce a line plot.",
         style = "font-size: 17px; text-align: center; color: black;"),
  tags$br(),
  
  fluidRow(
    column(3, selectInput("style", "Style of Alcohol", c("Beer", "Wine", "Spirits", "All Types"), selected = "All Types")),
    column(3, selectInput("state", "Select State(s)", c("All", states), multiple = TRUE, selected = "All")),
    column(3, selectInput("organization", "Type of Organization (For All States)",c("by States", "by Consumption"))),
    column(3, selectInput("order", "Sort Order", c("asc", "desc")))
  ),
  
  tags$br(),
  
  hr(),
  
  fluidRow(column(3, verbatimTextOutput("value"))),
  
  plotOutput("US_alc_Consumption", width = "auto", height = "800px"),
  tags$br()
)

# Define server logic
server <- function(input, output) {
  output$US_alc_Consumption <- renderPlot({
    
    if ("All" %in% input$state && length(input$state) == 1) {
      filtered_data <- switch(input$style,
                              "Beer" = alc %>% group_by(State) %>% summarise(total = mean(`Beer (Per capita consumption)`)),
                              "Wine" = alc %>% group_by(State) %>% summarise(total = mean(`Wine (Per capita consumption)`)),
                              "Spirits" = alc %>% group_by(State) %>% summarise(total = mean(`Spirits (Per capita consumption)`)),
                              "All Types" = alc %>% group_by(State) %>% summarise(total = mean(`All beverages (Per capita consumption)`))
      )
      
      top_states <- filtered_data %>% slice_max(n = 5, order_by = total)
      
      # how to order the data (either the State or the total consumption of the selected style of alcohol)
      if (input$organization == "by States") {
        if (input$order == "asc") {
          filtered_data <- filtered_data %>% mutate(State = factor(State, levels = rev(unique(State)))) %>% arrange(State)
        } else {   }
      } else {
        if (input$order == "asc") {
          filtered_data <- filtered_data %>% mutate(State = fct_reorder(State, total)) %>% arrange(total)
        } else {
          # now is reverse order
          filtered_data <- filtered_data %>% mutate(State = fct_reorder(State, -total)) %>% arrange(total)
        }
      }
      
      plt <- ggplot(filtered_data, aes(x = State, y = total, fill = ifelse(State %in% top_states$State, "Top 5", "Other"))) +
        geom_bar(stat = "identity") +
        scale_fill_manual(values = c("Top 5" = "orange", "Other" = "gray")) +
        theme_minimal() +
        labs(x = "State", y = input$style) +
        coord_flip() +
        labs(title = paste("US", input$style, "Consumption")) +
        theme(axis.title = element_text(size = 35)) +
        theme(axis.text.y = element_text(size = 12, face = ifelse(filtered_data$State %in% top_states$State, "bold.italic", "plain"))) +
        theme(axis.text.x = element_text(size = 20)) +
        theme(plot.title = element_text(hjust = 0.5, size = 45)) +
        guides(fill = guide_legend(title = "Top 5 States")) +
        theme(legend.title = element_text(size = 20)) +
        geom_text(aes(label = ifelse(State %in% top_states$State, round(total, 2), "")), hjust = -0.1, size = 5, color = "black")
      
    } else {
      
      filtered_data <- switch(input$style,
                              "Beer" = alc %>% filter(State %in% input$state) %>% 
                                group_by(State, Year) %>% 
                                summarise(total = mean(`Beer (Per capita consumption)`)),
                              "Wine" = alc %>% filter(State %in% input$state) %>% 
                                group_by(State, Year) %>% 
                                summarise(total = mean(`Wine (Per capita consumption)`)),
                              "Spirits" = alc %>% filter(State %in% input$state) %>% 
                                group_by(State, Year) %>% 
                                summarise(total = mean(`Spirits (Per capita consumption)`)),
                              "All Types" = alc %>% filter(State %in% input$state) %>% 
                                group_by(State, Year) %>% 
                                summarise(total = mean(`All beverages (Per capita consumption)`))
      )
      
      filtered_data <- filtered_data %>% arrange(State, desc(total))
      
      maxData <- filtered_data %>% group_by(State) %>% arrange(desc(total)) %>% slice(1)
      
      plt <- ggplot(filtered_data, aes(x = Year, y = total, color = State)) +
        geom_line() +
        theme_minimal() +
        labs(title = paste(paste(input$state, collapse = ", "), input$style, "Consumption"), x = "Year", y = input$style) +
        theme(axis.title = element_text(size = 30)) +
        theme(axis.text = element_text(size = 22)) +
        theme(plot.title = element_text(hjust = 0.5, size = 35)) +
        # increase the size of the legend
        theme(legend.title = element_text(size = 23)) +
        theme(legend.text = element_text(size = 16))
    }
    
    plt
    
  }, width = "auto", height = "auto") # allows for plots to resize with the window
}

# Run the application 
shinyApp(ui = ui, server = server)











