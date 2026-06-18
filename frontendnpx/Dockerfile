FROM node:20-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN npm run build

# Заставляем Next.js слушать внешние запросы от Nginx
ENV HOST=0.0.0.0
EXPOSE 3000

CMD ["npm", "start"]